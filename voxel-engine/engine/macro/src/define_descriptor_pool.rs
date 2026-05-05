use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{Attribute, Ident, Token, Visibility};

struct DefineDescriptorPoolInput {
    attrs: Vec<Attribute>,
    vis: Visibility,
    name: Ident,
}

impl Parse for DefineDescriptorPoolInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis: Visibility = input.parse()?;
        input.parse::<Token![struct]>()?;
        let name: Ident = input.parse()?;
        input.parse::<Token![;]>()?;

        Ok(DefineDescriptorPoolInput { attrs, vis, name })
    }
}

pub fn define_descriptor_pool_impl(input: TokenStream) -> TokenStream {
    let DefineDescriptorPoolInput { attrs, vis, name } =
        syn::parse_macro_input!(input as DefineDescriptorPoolInput);

    let expanded = quote! {
        #(#attrs)*
        #vis struct #name;

        impl crate::vulkan::resource_config::DescriptorPoolMarker for #name {
            fn id() -> &'static std::sync::atomic::AtomicUsize {
                static ID: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(usize::MAX);
                &ID
            }
        }
    };

    expanded.into()
}
