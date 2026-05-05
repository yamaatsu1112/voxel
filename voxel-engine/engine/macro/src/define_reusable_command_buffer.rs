use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{Attribute, Expr, Ident, Token, Visibility};

struct DefineReusableCommandBufferInput {
    attrs: Vec<Attribute>,
    vis: Visibility,
    name: Ident,
    count: Expr,
}

impl Parse for DefineReusableCommandBufferInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis: Visibility = input.parse()?;
        input.parse::<Token![struct]>()?;
        let name: Ident = input.parse()?;
        input.parse::<Token![;]>()?;

        let ident: Ident = input.parse()?;
        assert_eq!(ident, "count", "Expected 'count'");
        input.parse::<Token![=]>()?;
        let count: Expr = input.parse()?;
        input.parse::<Token![;]>()?;

        Ok(Self {
            attrs,
            vis,
            name,
            count,
        })
    }
}

pub fn define_reusable_command_buffer_impl(input: TokenStream) -> TokenStream {
    let DefineReusableCommandBufferInput {
        attrs,
        vis,
        name,
        count,
    } = syn::parse_macro_input!(input as DefineReusableCommandBufferInput);

    let ids_total_count = quote! {
        (#count) * crate::vulkan::vulkan_renderer_core::MAX_COMPUTE_FRAMES_IN_FLIGHT
    };

    let expanded = quote! {
        #(#attrs)*
        #vis struct #name;

        impl crate::vulkan::resource_config::ReusableCommandBufferMarker for #name {
            const COUNT: usize = #count;

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
