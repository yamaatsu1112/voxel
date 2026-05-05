use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{Attribute, Expr, Ident, Token, Visibility};

struct DefineBufferInput {
    attrs: Vec<Attribute>,
    vis: Visibility,
    name: Ident,
    size: syn::Path,
    lifetime: Ident,
    count: Expr,
    config: Expr,
}

impl Parse for DefineBufferInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis: Visibility = input.parse()?;
        input.parse::<Token![struct]>()?;
        let name: Ident = input.parse()?;
        input.parse::<Token![;]>()?;

        // size = <path>;
        let ident: Ident = input.parse()?;
        assert_eq!(ident, "size", "Expected 'size'");
        input.parse::<Token![=]>()?;
        let size: syn::Path = input.parse()?;
        input.parse::<Token![;]>()?;

        // lifetime = <ident>;
        let ident: Ident = input.parse()?;
        assert_eq!(ident, "lifetime", "Expected 'lifetime'");
        input.parse::<Token![=]>()?;
        let lifetime: Ident = input.parse()?;
        input.parse::<Token![;]>()?;

        // count = <expr>;
        let ident: Ident = input.parse()?;
        assert_eq!(ident, "count", "Expected 'count'");
        input.parse::<Token![=]>()?;
        let count: Expr = input.parse()?;
        input.parse::<Token![;]>()?;

        // config = <expr>;
        let ident: Ident = input.parse()?;
        assert_eq!(ident, "config", "Expected 'config'");
        input.parse::<Token![=]>()?;
        let config: Expr = input.parse()?;
        input.parse::<Token![;]>()?;

        Ok(DefineBufferInput {
            attrs,
            vis,
            name,
            size,
            lifetime,
            count,
            config,
        })
    }
}

pub fn define_buffer_impl(input: TokenStream) -> TokenStream {
    let DefineBufferInput {
        attrs,
        vis,
        name,
        size,
        lifetime,
        count,
        config,
    } = syn::parse_macro_input!(input as DefineBufferInput);

    let lifetime_str = lifetime.to_string();
    let ids_total_count: TokenStream2 = match lifetime_str.as_str() {
        "Persistent" => {
            quote! { #count }
        }
        "PerFrame" => {
            quote! { (#count) * crate::vulkan::vulkan_renderer_core::MAX_FRAMES_IN_FLIGHT }
        }
        _ => {
            return syn::Error::new(lifetime.span(), "Expected 'Persistent' or 'PerFrame'")
                .to_compile_error()
                .into();
        }
    };

    let expanded = quote! {
        #(#attrs)*
        #vis struct #name;

        impl crate::vulkan::resource_config::BufferMarker for #name {
            type Size = #size;
            type Lifetime = crate::vulkan::resource_lifetime::#lifetime;
            const COUNT: usize = #count;
            const CONFIG: crate::vulkan::resource_config::BufferConfig = #config;

            fn ids_slice() -> &'static [std::sync::atomic::AtomicUsize] {
                // SAFETY: AtomicUsize and usize have identical memory layout.
                static IDS: [std::sync::atomic::AtomicUsize; #ids_total_count] =
                    unsafe { std::mem::transmute([usize::MAX; #ids_total_count]) };
                &IDS
            }
        }
    };

    expanded.into()
}
